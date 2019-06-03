elifeLibrary {
    def commit

    stage 'Checkout', {
        checkout scm
        commit = elifeGitRevision()
    }

    node('containers-jenkins-plugin') {
        stage 'Build and run tests', {
            checkout scm
            try {
                sh "make IMAGE_TAG=${commit} ci-build-and-test"
            } finally {
                sh "make ci-clean"
            }
        }
    }

    elifeMainlineOnly {
        stage 'Merge to master', {
            elifeGitMoveToBranch commit, 'master'
        }
    }
}
